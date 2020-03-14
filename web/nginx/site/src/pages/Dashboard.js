import React from 'react';
import TopMenu from "../TopMenu";
import {
  Button,
  Container,
  Divider,
  Form,
  Grid,
  Header,
  Icon,
  Label,
  List,
  Message,
  Modal, Pagination,
  Popup,
  Search,
  Segment
} from "semantic-ui-react";
import Footer from "../Footer";
import {
  BASE_URL,
  cropText,
  deleteAgent,
  displayErrors,
  ensureSignedIn,
  fetchUsers,
  getJWT,
  hasJWT,
  refreshJWT
} from "../Util";
import _ from 'lodash'
import logo from "../icon.svg";
import {SemanticToastContainer, toast} from 'react-semantic-toasts';

// const initialState = {isLoading: false, results: [], value: ''};
// const source = ["blue", "red", "green"];

export default class Dashboard extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      creatingAgent: false,
      createAgentName: '',
      createAgentDescription: '',
      createAgentErrors: [],
      agents: [],
      agentParametersFileUpload: null,
      users: {},
      deleting: false,
      deletingAgent: null,
      pageSize: 5,
      currentPage: 1,
      agentCount: 0,
    };
    this.fileInputRef = React.createRef();
  }

  handleResultSelect = (e, {result}) => this.setState({value: result.title});

  handleSearchChange = (e, {value}) => {
    // this.setState({isLoading: true, value});
    //
    // setTimeout(() => {
    //   if (this.state.value.length < 1) return this.setState(initialState);
    //
    //   const re = new RegExp(_.escapeRegExp(this.state.value), 'i');
    //   const isMatch = (result) => re.test(result.title);
    //
    //   this.setState({
    //     isLoading: false,
    //     results: _.filter(source, isMatch),
    //   })
    // }, 300);
  };

  agentComponents = () => {
    let components = [];
    this.state.agents.forEach(agent => {
      if (!(agent.author in this.state.users)) {
        return;
      }

      let author = this.state.users[agent.author];

      components.push((
        <Grid.Column key={agent.id} className='sixteen wide'>
          <Segment textAlign='left'>
            <Label color='green' ribbon>
              Created by {author.username} ({author.first_name} {author.last_name})
            </Label>
            <br/>
            <span><h1>{agent.name}</h1></span>
            <br/>
            <span>{cropText(agent.description, 128)}</span>
            <br/>
            <Divider/>
            <Grid columns={2}>
              <Grid.Column>
                <Button
                  primary
                  download
                  href={agent.parameters}
                  icon='cloud download'
                  content='Download'
                  size='medium'
                />
                <Button
                  color='orange'
                  href={'/agent/' + agent.id}
                  icon='bars'
                  content='Details'
                  size='medium'
                />
              </Grid.Column>
              <Grid.Column>
                <div style={{textAlign: 'right'}}>
                  {this.deleteAgentModal(agent)}
                </div>
              </Grid.Column>
            </Grid>
          </Segment>
        </Grid.Column>
      ));
    });
    return components;
  };

  agentFileChange = (input) => {
    let file = input.current.files[0];
    if (!file.name.endsWith('.tar.gz')) {
      toast(
        {
          type: 'error',
          title: 'Bad file',
          description: <p>You must select a valid agent file</p>
        },
      );
      return;
    }
    this.setState({agentParametersFileUpload: file});
  };

  closeAgentCreation = () => {
    this.setState({
      creatingAgent: false,
      createAgentName: '',
      createAgentDescription: '',
      agentParametersFileUpload: null,
      createAgentErrors: [],
    });
  };

  createAgent = () => {
    let data = new FormData();
    data.append('name', this.state.createAgentName);
    data.append('description', this.state.createAgentDescription);
    data.append('parameters', this.state.agentParametersFileUpload);
    data.append('changeReason', 'Initial creation');

    fetch(BASE_URL + '/api/agents/', {
      method: 'POST',
      headers: {
        'Authorization': 'JWT ' + getJWT(),
      },
      body: data,
    }).then((response) => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 201) {
        console.error('Failed to create an agent: ', response);
        response.json().then(body => {
          this.setState({
            createAgentErrors: displayErrors(body['name'], body['description'])
          });
        });
        return;
      }

      this.closeAgentCreation();

      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Agent successfully created</p>
        },
      );

      this.fetchAgents(this.state.currentPage);
    });
  };

  unableToCreateAgentReasons = () => {
    let reasons = [];
    if (this.state.createAgentName === '') {
      reasons.push((<List.Item key='1'>The agent must have a name</List.Item>));
    }

    if (this.state.createAgentDescription === '') {
      reasons.push((<List.Item key='2'>The agent must have a description</List.Item>));
    }

    if (this.state.agentParametersFileUpload === null) {
      reasons.push((<List.Item key='3'>Agent training parameters must be provided</List.Item>));
    }

    return reasons;
  };

  componentDidMount = () => {
    ensureSignedIn();
    this.fetchAgents(this.state.currentPage);
  };

  fetchAgents = (url) => {
    if (!hasJWT()) {
      return;
    }

    fetch(BASE_URL + '/api/agents/?page_size=' + this.state.pageSize + '&page=' + this.state.currentPage, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        response.text().then(body => {
          console.error("Unable to fetch agents: ", body);
        });
        return;
      }

      response.json().then(body => {
        this.setState({
          agents: body.results,
          agentCount: body.count,
        });
        let userURIs = new Set();
        this.state.agents.forEach(agent => {
          userURIs.add(agent['author']);
        });
        fetchUsers(userURIs, users => {
          this.setState({users: users})
        });
      });
    });
  };

  deleteAgentModal = (agent) => (
    <Modal open={this.state.deleting}
           trigger={
             <Button icon onClick={() => {
               this.setState({deleting: true})
             }} color='red'>
               <Icon name='cancel'/>
             </Button>
           }
           basic
           size='small'>
      <Header icon='cancel' content='Delete Agent'/>
      <Modal.Content>
        <b>Warning:</b> Deleting the agent <b>"{agent.name}"</b>, are you sure you want to do this?
      </Modal.Content>
      <Modal.Actions>
        <Button basic color='grey' inverted onClick={() => this.setState({deleting: false})}>
          Cancel
        </Button>
        <Button
          color='red'
          inverted
          onClick={() => deleteAgent(agent.id, () => window.location = '/dashboard')}
        >
          Delete
        </Button>
      </Modal.Actions>
    </Modal>
  );

  createAgentModal = () => (
    <Modal open={this.state.creatingAgent}
           trigger={
             <Button icon positive onClick={() =>
               this.setState({creatingAgent: true})
             }>
               <Icon name='plus'/>{} Create a new agent
             </Button>
           }
           basic
           size='small'>
      <Header icon='add square' content='Create Agent'/>
      <Modal.Content>
        <p>Create an agent to backup on our cloud service.</p>
        <Form.Input
          fluid
          required
          icon="tag"
          iconPosition="left"
          placeholder="Name"
          onChange={event => this.setState({createAgentName: event.target.value})}
        />
        <p/>
        <Form.Input
          fluid
          required
          icon="pencil"
          iconPosition="left"
          placeholder="Description"
          onChange={event => this.setState({createAgentDescription: event.target.value})}
        />
        <p/>
        <Popup
          flowing
          position='right center'
          trigger={
            <Button
              content={this.state.agentParametersFileUpload === null ? "Choose File" : this.state.agentParametersFileUpload.name}
              labelPosition="left"
              icon="file"
              color='green'
              onClick={() => this.fileInputRef.current.click()}
            />
          }>
          Find your agent parameter files
          under: <br/><code>/home/&lt;username&gt;/.agents/&lt;name&gt;-&lt;timestamp&gt;.tar.gz</code>
        </Popup>
        <input
          ref={this.fileInputRef}
          type="file"
          hidden
          onChange={(event) => {
            this.agentFileChange(this.fileInputRef);
          }}
        />
        <Message
          error
          header='Cannot create agent'
          list={this.state.createAgentErrors}
          hidden={this.state.createAgentErrors.length === 0}
        />
      </Modal.Content>
      <Modal.Actions>
        <Button basic color='grey' inverted onClick={() => this.closeAgentCreation()}>
          <Icon name='remove'/> Cancel
        </Button>
        <Popup
          flowing
          position='bottom right'
          disabled={this.unableToCreateAgentReasons().length === 0}
          trigger={
            <span>
              <Button
                color='green'
                inverted
                disabled={this.unableToCreateAgentReasons().length !== 0}
                onClick={() => this.createAgent()}
              >
                <Icon name='checkmark'/> Create
              </Button>
            </span>
          }>
          <List bulleted>
            {this.unableToCreateAgentReasons()}
          </List>
        </Popup>
      </Modal.Actions>
    </Modal>
  );

  setPage = (event, {activePage}) => {
    this.setState({currentPage: Math.ceil(activePage)}, this.fetchAgents);
  };

  getPagination() {
    let totalPages = Math.ceil(this.state.agentCount / this.state.pageSize);
    if (totalPages <= 1) {
      return;
    }

    return <Pagination
      activePage={this.state.currentPage}
      totalPages={totalPages}
      onPageChange={this.setPage}
    />;
  }

  render = () => (
    <div className="SitePage">
      <TopMenu/>
      <SemanticToastContainer position='bottom-right'/>
      <Container className="SiteContents" style={{marginTop: '80px', marginBottom: '32px'}}>
        <Header as="h2" color="teal" textAlign="center">
          <img src={logo} alt="logo" className="image"/>{" "}
          Agents
        </Header>
        <Divider/>
        <Segment basic textAlign='center'>
          <Grid columns={2} relaxed='very'>
            <Grid.Column>
              <Search
                placeholder='Search agents'
                loading={this.state.isLoading}
                onResultSelect={this.handleResultSelect}
                onSearchChange={_.debounce(this.handleSearchChange, 500, {
                  leading: true,
                })}
                results={this.state.results}
                value={this.state.value}
              />
            </Grid.Column>
            <Grid.Column>
              {this.createAgentModal()}
            </Grid.Column>
          </Grid>
          <Divider vertical>Or</Divider>
        </Segment>
        {this.getPagination()}
        <Grid style={{marginTop: '16px', marginBottom: '16px'}}>
          {this.agentComponents()}
        </Grid>
        {this.getPagination()}
      </Container>
      < Footer/>
    </div>
  );
}
