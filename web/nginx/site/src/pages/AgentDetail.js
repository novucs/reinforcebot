import React, {Component} from "react";
import TopMenu from "../TopMenu";
import {
  Breadcrumb,
  Button,
  Container,
  Divider,
  Form,
  Grid,
  Header,
  Icon,
  List,
  Loader,
  Modal,
  Pagination,
  Popup,
  Segment,
  Table
} from "semantic-ui-react";
import Footer from "../Footer";
import logo from "../icon.svg";
import {BASE_URL, deleteAgent, ensureSignedIn, fetchUsers, getJWT, hasJWT, refreshJWT} from "../Util";
import Moment from 'moment';
import {SemanticToastContainer, toast} from "react-semantic-toasts";

export default class AgentDetail extends Component {
  constructor(props) {
    super(props);
    this.state = {
      agent: null,
      users: {},
      historyPageSize: 10,
      historyPageCount: 0,
      currentHistoryPage: 1,
      editingName: false,
      editingDescription: false,
      updatingModal: false,
      agentParametersFileUpload: null,
      deleting: false,
      deleteValue: false,
    };
    this.fileInputRef = React.createRef();
  }

  componentDidMount = () => {
    ensureSignedIn();
    this.fetchAgent();
  };

  fetchAgent = () => {
    if (!hasJWT()) {
      return;
    }

    fetch(BASE_URL + '/api/agents/' + this.props.match.params.id + '/', {
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
          console.error("Unable to fetch agent: ", body);
        });
        return;
      }

      response.json().then(agent => {
        this.setState({
          agent: agent,
          historyPageCount: Math.ceil(agent.history.length / this.state.historyPageSize),
        });

        let userURIs = [agent.author];
        agent.history.forEach(h => {
          userURIs.push(BASE_URL + '/api/auth/users/' + h.history_user_id + '/');
        });

        fetchUsers(userURIs, users => {
          this.setState({users: users});
        });
      });
    });
  };

  agentHistory = () => {
    let history = [];

    let startIndex = (this.state.currentHistoryPage - 1) * this.state.historyPageSize;
    let stopIndex = Math.min(
      startIndex + this.state.historyPageSize,
      this.state.agent.history.length
    );

    for (let i = startIndex; i < stopIndex; i++) {
      let item = this.state.agent.history[i];
      if (!(item.history_user_id in this.state.users)) {
        return;
      }
      let user = this.state.users[item.history_user_id];
      Moment.locale('en');
      history.push((
        <Table.Row>
          <Table.Cell>
            <a download href={BASE_URL + '/api/media/' + item.parameters}>
              <Icon name='cloud download'/>
            </a>
          </Table.Cell>
          <Table.Cell>
            {user.username}
          </Table.Cell>
          <Table.Cell>{item.history_change_reason}</Table.Cell>
          <Table.Cell>
            <Popup
              inverted
              hoverable
              position='bottom center'
              content={Moment(item.history_date).format('LLLL')}
              trigger={<span>{Moment(item.history_date).fromNow()}</span>}
            />
          </Table.Cell>
        </Table.Row>
      ));
    }

    // this.state.agent.history.forEach(item => {
    // });
    return history;
  };

  closeEditWindow = () => {
    this.setState({
      editingName: false,
      editingDescription: false,
      updatingModal: false,
      agentParametersFileUpload: null,
      deleting: false,
      deleteValue: false,
    });
  };

  editAgent = fieldName => {
    if (!hasJWT() || this.state.agent[fieldName] === '') {
      return;
    }

    let body = {};
    body[fieldName] = this.state.agent[fieldName];
    body['changeReason'] = 'Updated ' + fieldName;

    fetch(BASE_URL + '/api/agents/' + this.state.agent.id + '/', {
      method: 'PATCH',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
      body: JSON.stringify(body)
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        response.text().then(body => {
          console.error("Unable to update agent: ", body);
        });
        return;
      }

      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Agent successfully updated</p>
        },
      );
      this.closeEditWindow();
      this.fetchAgent();
    });
  };

  editNameModal = () => (
    <Modal open={this.state.editingName}
           trigger={
             <Button
               fluid
               style={{marginTop: '5px'}}
               icon='tag'
               content='Edit Name'
               onClick={() => this.setState({editingName: true})}
             />
           }
           basic
           size='small'>
      <Header icon='tag' content='Editing Agent Name'/>
      <Modal.Content>
        <Form.Input
          fluid
          required
          icon="tag"
          iconPosition="left"
          placeholder="Name"
          defaultValue={this.state.agent.name}
          onChange={event => this.setState({
            agent: {...this.state.agent, name: event.target.value}
          })}
        />
      </Modal.Content>
      <Modal.Actions>
        <Button basic color='grey' inverted onClick={() => this.closeEditWindow()}>
          <Icon name='remove'/> Cancel
        </Button>
        <Button
          color='green'
          inverted
          disabled={this.state.agent.name === ''}
          onClick={() => this.editAgent('name')}
        >
          <Icon name='checkmark'/> Submit
        </Button>
      </Modal.Actions>
    </Modal>
  );

  editDescriptionModal = () => (
    <Modal open={this.state.editingDescription}
           trigger={
             <Button
               fluid
               style={{marginTop: '5px'}}
               icon='pencil'
               content='Edit Description'
               onClick={() => this.setState({editingDescription: true})}
             />
           }
           basic
           size='small'>
      <Header icon='pencil' content='Editing Agent Description'/>
      <Modal.Content>
        <Form.TextArea
          style={{width: '100%'}}
          rows={10}
          required
          placeholder="Description"
          defaultValue={this.state.agent.description}
          onChange={event => this.setState({
            agent: {...this.state.agent, description: event.target.value}
          })}
        />
      </Modal.Content>
      <Modal.Actions>
        <Button basic color='grey' inverted onClick={() => this.closeEditWindow()}>
          <Icon name='remove'/> Cancel
        </Button>
        <Button
          color='green'
          inverted
          disabled={this.state.agent.description === ''}
          onClick={() => this.editAgent('description')}
        >
          <Icon name='checkmark'/> Submit
        </Button>
      </Modal.Actions>
    </Modal>
  );

  updateParameters = () => {
    if (!hasJWT() || this.state.agentParametersFileUpload === null) {
      return;
    }

    let data = new FormData();
    data.append('parameters', this.state.agentParametersFileUpload);
    data.append('changeReason', 'Updated parameters');

    fetch(BASE_URL + '/api/agents/' + this.state.agent.id + '/', {
      method: 'PATCH',
      headers: {
        'Authorization': 'JWT ' + getJWT(),
      },
      body: data,
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        response.text().then(body => {
          console.error("Unable to update agent: ", body);
        });
        return;
      }

      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Agent successfully updated</p>
        },
      );
      this.closeEditWindow();
      this.fetchAgent();
    });
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

  updateParametersModal = () => (
    <Modal open={this.state.updatingModal}
           trigger={
             <Button
               fluid
               style={{marginTop: '5px'}}
               color='yellow'
               icon='cog'
               content='Update Model'
               onClick={() => this.setState({updatingModal: true})}
             />
           }
           basic
           size='small'>
      <Header icon='upload' content='Update Agent Modal'/>
      <Modal.Content>
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
      </Modal.Content>
      <Modal.Actions>
        <Button basic color='grey' inverted onClick={() => this.closeEditWindow()}>
          <Icon name='remove'/> Cancel
        </Button>
        <Button
          color='green'
          inverted
          disabled={this.state.agentParametersFileUpload === null}
          onClick={() => this.updateParameters()}
        >
          <Icon name='checkmark'/> Upload
        </Button>
      </Modal.Actions>
    </Modal>
  );

  deleteAgentModal = () => (
    <Modal open={this.state.deleting}
           trigger={
             <Button
               fluid
               style={{marginTop: '5px'}}
               color='red'
               icon='cancel'
               content='Delete'
               onClick={() => this.setState({deleting: true})}
             />
           }
           basic
           size='small'>
      <Header icon='cancel' content='Delete Agent'/>
      <Modal.Content>
        <b>Warning:</b> Deleting the agent <b>"{this.state.agent.name}"</b>, are you sure you want to do this?
      </Modal.Content>
      <Modal.Actions>
        <Button basic color='grey' inverted onClick={() => this.closeEditWindow()}>
          Cancel
        </Button>
        <Button
          color='red'
          inverted
          onClick={() => deleteAgent(this.state.agent.id, () => window.location = '/dashboard')}
        >
          Delete
        </Button>
      </Modal.Actions>
    </Modal>
  );

  descriptionLines = () => {
    let lines = [];
    let i = 0;
    this.state.agent.description.split('\n').forEach(line => {
      i += 1;
      lines.push((
        <List.Item key={'line-' + i}>{line}</List.Item>
      ))
    });
    return lines;
  };

  setHistoryPage = (event, {activePage}) => {
    this.setState({currentHistoryPage: Math.ceil(activePage)});
  };

  agentContent = () => (
    <div>
      <Header as="h2" color="teal" textAlign="center">
        <img src={logo} alt="logo" className="image"/>{" "}
        {this.state.agent.name}
      </Header>
      <Grid style={{marginBottom: '32px'}}>
        <Grid.Column className='eleven wide'>
          <Segment textAlign='left'>
            <Breadcrumb icon='right angle' sections={[
              {key: 'Dashboard', content: 'Dashboard', href: '/dashboard'},
              {key: 'Agent', content: 'Agent', active: true},
            ]}/>
            <Divider/>
            <List className='large text'>
              {this.descriptionLines()}
            </List>
          </Segment>
          <div style={{textAlign: 'left'}}>
            <Table celled striped>
              <Table.Header>
                <Table.Row>
                  <Table.HeaderCell colSpan='4'>History</Table.HeaderCell>
                </Table.Row>
              </Table.Header>
              <Table.Body>
                {this.agentHistory()}
              </Table.Body>
            </Table>
            <Grid>
              <Grid.Column textAlign='center'>
                <Pagination
                  defaultActivePage={this.state.currentHistoryPage}
                  totalPages={this.state.historyPageCount}
                  onPageChange={this.setHistoryPage}
                />
              </Grid.Column>
            </Grid>
          </div>
        </Grid.Column>
        <Grid.Column className='five wide'>
          <Segment>
            <Header as='h4'>Options</Header>
            <Divider/>
            <Button
              fluid
              primary
              download
              href={this.state.agent.parameters}
              icon='cloud download'
              content='Download'
            />
            {this.editNameModal()}
            {this.editDescriptionModal()}
            {this.updateParametersModal()}
            {this.deleteAgentModal()}
          </Segment>
        </Grid.Column>
      </Grid>
    </div>
  );

  render = () => (
    <div className="SitePage">
      <TopMenu/>
      <SemanticToastContainer position='bottom-right'/>
      <Container className="SiteContents" style={{marginTop: '80px'}}>
        {this.state.agent !== null ? this.agentContent() : (
          <Loader/>
        )}
      </Container>
      < Footer/>
    </div>
  );
}