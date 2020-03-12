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
  Label, Message,
  Modal,
  Popup,
  Search,
  Segment
} from "semantic-ui-react";
import Footer from "../Footer";
import {BASE_URL, displayErrors, ensureSignedIn, getJWT, hasJWT, refreshJWT} from "../Util";
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
      createAgentErrors: [
        // 'Agent already exists',
      ],
      agents: [],
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

  deleteAgent(id) {
    if (hasJWT()) {
      fetch(BASE_URL + '/api/agents/' + id + '/', {
        method: 'DELETE',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Authorization': 'JWT ' + getJWT(),
        },
      }).then(response => {
        if (response.status === 401) {
          refreshJWT();
          // todo: refresh on success?
          return;
        }

        if (response.status !== 204) {
          response.text().then(body => {
            console.error("Unable to delete agent: ", response);
          });
          return;
        }

        this.fetchAgents();
      });
    }
  }

  agentComponents() {
    let components = [];
    this.state.agents.forEach(agent => {
      components.push((
        <Grid.Column key={agent['id']} className='eight wide'>
          <Segment textAlign='left'>
            <Label color='green' ribbon>
              Author
            </Label>
            <br/>
            {/*<span><h1>Agent #{agent['id']}</h1></span>*/}
            <span><h1>{agent['name']}</h1></span>
            <br/>
            <span>{agent['author']}</span>
            <div style={{textAlign: 'right'}}>
              <Button size='tiny' icon onClick={() => {
                this.deleteAgent(agent['id'])
              }} color='red'>
                <Icon name='cancel'/>
              </Button>
            </div>
          </Segment>
        </Grid.Column>
      ));
    });
    return components;
  }

  fileChange(event) {
    console.log('File change: ', event);
  }

  createAgent() {
    fetch(BASE_URL + '/api/agents/', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
      body: JSON.stringify({
        name: this.state.createAgentName,
        description: this.state.createAgentDescription,
      }),
    }).then(response => {
      if (response.status !== 201) {
        response.json().then(body => {
          this.setState({
            createAgentErrors: displayErrors(body['name'], body['description'])
          });
        });
        return;
      }
      this.setState({creatingAgent: false});
      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Agent successfully created</p>
        },
      );
      this.fetchAgents();
    });
  }

  canCreateAgent() {
    return this.state.createAgentName !== '' && this.state.createAgentDescription !== '';
  }

  componentDidMount() {
    ensureSignedIn();
    this.fetchAgents();
  }

  fetchAgents() {
    if (hasJWT()) {
      fetch(BASE_URL + '/api/agents/', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Authorization': 'JWT ' + getJWT(),
        },
      }).then(response => {
        if (response.status === 401) {
          refreshJWT();
          // todo: refresh on success?
          return;
        }

        if (response.status !== 200) {
          response.text().then(body => {
            console.error("Unable to fetch agents: ", body);
          });
          return;
        }

        response.json().then(body => {
          this.setState({agents: body});
        });
      });
    }
  }

  render() {
    return (
      <div className="SitePage">
        <TopMenu/>
        <SemanticToastContainer position='bottom-right'/>
        <Container className="SiteContents" style={{marginTop: '80px'}}>
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
                      onKeyDown={this.keyPress}
                      onChange={event => this.setState({createAgentName: event.target.value})}
                    />
                    <p/>
                    <Form.Input
                      fluid
                      required
                      icon="pencil"
                      iconPosition="left"
                      placeholder="Description"
                      onKeyDown={this.keyPress}
                      onChange={event => this.setState({createAgentDescription: event.target.value})}
                    />
                    <p/>
                    <Popup
                      flowing
                      position='right center'
                      trigger={
                        <Button
                          content="Choose File"
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
                      onChange={this.fileChange}
                    />
                    <Message
                      error
                      header='Cannot create agent'
                      list={this.state.createAgentErrors}
                      hidden={this.state.createAgentErrors.length === 0}
                    />
                  </Modal.Content>
                  <Modal.Actions>
                    <Button basic color='red' inverted onClick={() => this.setState({
                      createAgentName: '',
                      createAgentDescription: '',
                      creatingAgent: false
                    })}>
                      <Icon name='remove'/> Cancel
                    </Button>
                    <Button
                      color='green'
                      inverted
                      disabled={!this.canCreateAgent()}
                      onClick={() => this.createAgent()}
                    >
                      <Icon name='checkmark'/> Create
                    </Button>
                  </Modal.Actions>
                </Modal>
              </Grid.Column>
            </Grid>
            <Divider vertical>Or</Divider>
          </Segment>
          <Grid style={{marginTop: '16px', marginBottom: '16px'}}>
            {this.agentComponents()}
          </Grid>
        </Container>
        < Footer/>
      </div>
    );
  }
}
